import React from 'react';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ImageLab Caught UI Exception:', error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '3rem 2rem', maxWidth: '800px', margin: '2rem auto' }}>
          <div className="section-card" style={{ borderColor: 'var(--accent-rose)', borderLeft: '4px solid var(--accent-rose)' }}>
            <h3 style={{ color: 'var(--accent-rose)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span>⚠️</span>
              <span>An unexpected display error occurred</span>
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '1rem' }}>
              The module encountered a rendering exception. Your workspace and settings remain intact.
            </p>
            {this.state.error && (
              <pre style={{
                background: '#06090e',
                padding: '0.85rem 1rem',
                borderRadius: '6px',
                border: '1px solid var(--border-medium)',
                color: '#f87171',
                fontSize: '0.8rem',
                overflowX: 'auto',
                marginBottom: '1.25rem'
              }}>
                {this.state.error.toString()}
              </pre>
            )}
            <div style={{ display: 'flex', gap: '0.75rem' }}>
              <button
                type="button"
                className="btn btn-primary"
                onClick={this.handleReset}
              >
                Reload Module
              </button>
              <button
                type="button"
                className="btn btn-outline"
                onClick={() => this.setState({ hasError: false })}
              >
                Attempt Recovery
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
